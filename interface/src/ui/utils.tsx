import * as React from "react";
import { cva as cvaOriginal, cx as cxOriginal, type VariantProps } from "class-variance-authority";

export { VariantProps };
export const cva = cvaOriginal;
export const cx = cxOriginal;

type ElementType = keyof React.JSX.IntrinsicElements;

type TailwindFactory = {
  [K in ElementType]: (
    strings: TemplateStringsArray,
    ...values: (string | number | undefined | null | false)[]
  ) => React.ForwardRefExoticComponent<
    React.PropsWithoutRef<React.JSX.IntrinsicElements[K]> & React.RefAttributes<HTMLElement>
  >;
} & {
  <T extends React.ComponentType<any>>(
    component: T
  ): (
    strings: TemplateStringsArray,
    ...values: (string | number | undefined | null | false)[]
  ) => React.ForwardRefExoticComponent<
    React.PropsWithoutRef<React.ComponentPropsWithoutRef<T>> & React.RefAttributes<any>
  >;
};

function twFactory<T extends ElementType | React.ComponentType<any>>(type: T) {
  return (
    strings: TemplateStringsArray,
    ...values: (string | number | undefined | null | false)[]
  ): React.ForwardRefExoticComponent<any> => {
    const baseClasses = strings.reduce((acc, str, i) => {
      const value = values[i];
      if (value === undefined || value === null || value === false) {
        return acc + str;
      }
      return acc + str + String(value);
    }, "");

    return React.forwardRef((props: any, ref) => {
      const { className, ...rest } = props;
      const finalClassName = cx(baseClasses, className);
      
      if (typeof type === "string") {
        return React.createElement(type, {
          ref,
          className: finalClassName,
          ...rest,
        });
      }
      
      return React.createElement(type, {
        ref,
        className: finalClassName,
        ...rest,
      });
    }) as any;
  };
}

export const tw = new Proxy((() => {}) as unknown as TailwindFactory, {
  get(_, property: string) {
    return twFactory(property as ElementType);
  },
  apply(_, __, [element]: [React.ComponentType<any>]) {
    return twFactory(element);
  },
});

export function cn(...inputs: (string | number | undefined | null | false)[]) {
  return cx(inputs);
}
