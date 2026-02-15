"use client";

import * as React from "react";
import { useFormContext, Controller } from "react-hook-form";
import { TextArea, type TextAreaProps, Label } from "../Input";
import { ErrorMessage } from "./Form";
import { cx } from "../utils";

export interface TextAreaFieldProps extends Omit<TextAreaProps, "name"> {
  name: string;
  label?: string;
  description?: string;
  className?: string;
}

export const TextAreaField = React.forwardRef<
  HTMLTextAreaElement,
  TextAreaFieldProps
>(({ name, label, description, className, ...props }, ref) => {
  const { control, formState } = useFormContext();
  const error = formState.errors[name]?.message as string | undefined;

  return (
    <Controller
      name={name}
      control={control}
      render={({ field }) => (
        <div className={cx("space-y-1.5", className)}>
          {label && <Label htmlFor={name}>{label}</Label>}
          <TextArea
            {...field}
            {...props}
            ref={ref}
            id={name}
            error={!!error}
          />
          {description && !error && (
            <p className="text-xs text-ink-dull">{description}</p>
          )}
          {error && <ErrorMessage error={error} />}
        </div>
      )}
    />
  );
});

TextAreaField.displayName = "TextAreaField";
