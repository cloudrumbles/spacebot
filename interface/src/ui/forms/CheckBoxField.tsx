"use client";

import * as React from "react";
import { useFormContext, Controller } from "react-hook-form";
import { CheckboxField, type CheckboxProps } from "../Checkbox";
import { ErrorMessage } from "./Form";
import { cx } from "../utils";

export interface CheckBoxFieldProps extends Omit<CheckboxProps, "name" | "checked" | "onCheckedChange"> {
  name: string;
  label?: React.ReactNode;
  description?: string;
  className?: string;
}

export const CheckBoxField: React.FC<CheckBoxFieldProps> = ({
  name,
  label,
  description,
  className,
  ...props
}) => {
  const { control, formState } = useFormContext();
  const error = formState.errors[name]?.message as string | undefined;

  return (
    <Controller
      name={name}
      control={control}
      render={({ field }) => (
        <div className={cx("space-y-1.5", className)}>
          <CheckboxField
            label={label || name}
            description={description}
            checkboxProps={{
              ...props,
              checked: field.value,
              onCheckedChange: field.onChange,
            }}
          />
          {error && <ErrorMessage error={error} />}
        </div>
      )}
    />
  );
};

CheckBoxField.displayName = "CheckBoxField";
